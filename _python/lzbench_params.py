#!/usr/bin/env python

from __future__ import absolute_import, division, print_function


class LzbenchPreproc(object):
    __slots__ = 'name cmd_line_arg'.split()

    def __init__(self, name, cmd_line_arg):
        self.name = name
        self.cmd_line_arg = cmd_line_arg


_ALL_LZBENCH_PREPROCS = [
    LzbenchPreproc('Delta', '-F1'),
    LzbenchPreproc('DoubleDelta', '-F2'),
    LzbenchPreproc('FIRE', '-F3'),
    LzbenchPreproc('DynamicDelta', '-F4'),
    # TODO add in bitshuf/byteshuf, sprintz bitpack, etc
]
