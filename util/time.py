# -*- coding: utf-8 -*-
def hhmmss(seconds):
    return '{:02}:{:02}:{:02}'.format(int(seconds//3600), int(seconds%3600//60), int(seconds%60))
