#ifndef LED_H
#define LED_H

#include "config.h"
void init_led();
void set_led_duty(int value);
void set_debug_led(int value);
#endif