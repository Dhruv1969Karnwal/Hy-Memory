# -*- coding: utf-8 -*-
"""Backwards compatibility: use pipeline_log_writer.PipelineLogWriter instead."""

from .pipeline_log_writer import PipelineLogWriter, TraceFileWriter

__all__ = ["PipelineLogWriter", "TraceFileWriter"]
