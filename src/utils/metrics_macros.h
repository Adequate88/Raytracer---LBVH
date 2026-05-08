#pragma once

#ifdef EVALUATE

#include "metrics.h"
#define METRIC_SET_VALUE(name, value) Metrics::get().set_value(name, value)
#define METRIC_INCREMENT(name) Metrics::get().increment(name)
#define METRIC_READ(name) Metrics::get().read(name)
#define METRIC_START_TIME(name) Metrics::get().start_time(name)
#define METRIC_END_TIME(name) Metrics::get().end_time(name)
//#define METRIC_EXPORT(filename) Metrics::get().export(filename)

#else

#define METRIC_SET_VALUE(name, value) ((void)0)
#define METRIC_INCREMENT(name) ((void)0)
#define METRIC_READ(name) ((float)0)
#define METRIC_START_TIME(name) ((void)0)
#define METRIC_END_TIME(name) ((void)0)
#define METRIC_EXPORT(filename) ((void)0)

#endif


