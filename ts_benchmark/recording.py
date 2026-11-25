# -*- coding: utf-8 -*-

from __future__ import absolute_import

import io
import itertools
import logging
import math
import numbers
import os
import os.path
import csv
import tarfile
import tempfile
from io import StringIO
from typing import List, Optional

import pandas as pd
from pandas.errors import ParserError

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.compress import (
    get_compress_method_from_ext,
    decompress,
    compress,
    get_compress_file_ext,
)
from ts_benchmark.utils.get_file_name import get_unique_file_suffix

logger = logging.getLogger(__name__)


CSV_WRITE_OPTIONS = {
    "index": False,
    "quoting": csv.QUOTE_ALL,
    "doublequote": True,
    "escapechar": "\x1e",
}

CSV_READ_OPTIONS = {"escapechar": "\x1e"}


def _stream_csv_value(handle, value, chunk_size: int = 1024 * 1024) -> None:
    is_missing = value is None or (
        isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isnan(value)
    )
    handle.write('"')
    if hasattr(value, "iter_text_chunks"):
        chunks = value.iter_text_chunks(chunk_size)
    else:
        text = "" if is_missing else str(value)
        chunks = (
            text[start : start + chunk_size]
            for start in range(0, len(text), chunk_size)
        )
    for chunk in chunks:
        chunk = chunk.replace("\x1e", "\x1e\x1e").replace('"', '""')
        handle.write(chunk)
    handle.write('"')


def _cleanup_stream_values(result_df: pd.DataFrame) -> None:
    seen = set()
    for value in result_df.to_numpy(dtype=object).flat:
        cleanup = getattr(value, "cleanup", None)
        if cleanup is not None and id(value) not in seen:
            seen.add(id(value))
            cleanup()


def _write_record_csv_stream(result_df: pd.DataFrame, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8", newline="") as handle:
        rows = itertools.chain(
            (result_df.columns,), result_df.itertuples(index=False, name=None)
        )
        for row in rows:
            for index, value in enumerate(row):
                if index:
                    handle.write(",")
                _stream_csv_value(handle, value)
            handle.write("\n")


def read_record_file(fn: str) -> pd.DataFrame:
    """
    Reads a single record file.

    The format of the file is currently determined by the extension name.

    :param fn: Path to the record file.
    :return: Benchmarking records in DataFrame format.
    """
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    if compress_method is None:
        return pd.read_csv(fn, **CSV_READ_OPTIONS)
    else:
        with open(fn, "rb") as fh:
            data = fh.read()
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v.decode("utf8")), **CSV_READ_OPTIONS))
        return pd.concat(ret, axis=0)


def write_record_file(
    result_df: pd.DataFrame,
    file_path: str,
    compress_method: Optional[str] = None,
) -> str:
    """
    Write to a single record file.

    :param result_df: Benchmarking records in DataFrame format.
    :param file_path: Path to the record file to save.
    :param compress_method: The format used to compress the record file, if None is given,
        no compression is applied.
    :return: Path to the record file written.
    """
    try:
        return _write_record_file_impl(result_df, file_path, compress_method)
    finally:
        _cleanup_stream_values(result_df)


def _write_record_file_impl(
    result_df: pd.DataFrame,
    file_path: str,
    compress_method: Optional[str] = None,
) -> str:
    if compress_method == "gz":
        archive_path = f"{file_path}.{get_compress_file_ext(compress_method)}"
        output_dir = os.path.dirname(archive_path)
        csv_fd, csv_temp_path = tempfile.mkstemp(
            prefix="record_", suffix=".csv", dir=output_dir
        )
        archive_fd, archive_temp_path = tempfile.mkstemp(
            prefix="record_", suffix=".tar.gz.tmp", dir=output_dir
        )
        os.close(csv_fd)
        os.close(archive_fd)
        try:
            _write_record_csv_stream(result_df, csv_temp_path)
            with tarfile.open(
                archive_temp_path, mode="w:gz", compresslevel=1
            ) as archive:
                archive.add(csv_temp_path, arcname=os.path.basename(file_path))
            os.replace(archive_temp_path, archive_path)
        finally:
            for temp_path in (csv_temp_path, archive_temp_path):
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
        file_path = archive_path
    elif compress_method is not None:
        buf = io.StringIO()
        result_df.to_csv(buf, **CSV_WRITE_OPTIONS)
        write_data = compress(
            {os.path.basename(file_path): buf.getvalue()}, method=compress_method
        )
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"

        with open(file_path, "wb") as fh:
            fh.write(write_data)
    else:
        result_df.to_csv(file_path, **CSV_WRITE_OPTIONS)

    return file_path


def load_record_data(
    record_files: List[str], drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads benchmarking records from multiple record files.

    :param record_files: The list of paths to the record files. Each item in the list can either
        be the path to a directory or a file. If it is a path to a directory, then all record files
        in the directory are loaded; Otherwise, the file specified by the path is loaded.
    :param drop_columns: The columns to drop during loading.
        This parameter is mainly used to save memory.
    :return: The loaded benchmarking records in DataFrame format.
    """
    record_files = itertools.chain.from_iterable(
        [
            [fn] if not os.path.isdir(fn) else find_record_files(fn)
            for fn in record_files
        ]
    )

    ret = []
    for fn in record_files:
        logger.info("loading log file %s", fn)
        try:
            cur_record = read_record_file(fn)
            if drop_columns:
                cur_record = cur_record.drop(columns=drop_columns)
            ret.append(cur_record)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def find_record_files(directory: str) -> List[str]:
    """
    Finds records files in a directory.

    :param directory: The path to the directory.
    :return: The list of file paths to the record files that are found in the give directory.
    """
    record_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                record_files.append(os.path.join(root, file))
    return record_files


def save_log(
    result_df: pd.DataFrame, save_path, file_prefix: str, compress_method: str = "gz"
) -> str:
    """
    Save log data.

    Save the evaluation results, model hyperparameters, model evaluation configuration, and model name to a log file.

    :param result_df: Benchmarking records in DataFrame format.
    :param save_path: Path to the directory where the records are saved.
    :param file_prefix: Prefix of the file name to save the records.
    :param compress_method: The compression method for the output file.
    :return: The path to the output file.
    """
    if result_df["log_info"].any():
        error_itr = filter(None, result_df["log_info"])
        for error in itertools.islice(error_itr, 3):
            logger.info(error)
        if any(error_itr):
            logger.info(
                "-------------More error messages can be found in the record files!-------------"
            )

    if save_path is not None:
        result_path = (
            os.path.join(ROOT_PATH, "result", save_path)
            if not os.path.isabs(save_path)
            else save_path
        )
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    record_filename = file_prefix + get_unique_file_suffix()
    file_path = os.path.join(result_path, record_filename)

    return write_record_file(result_df, file_path, compress_method)
