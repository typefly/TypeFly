#include <stdio.h>
#include "driver/gpio.h"
#include "hal/gpio_types.h"

void app_main(void)
{
    gpio_set_direction(GPIO_NUM_33, GPIO_MODE_OUTPUT);
    gpio_set_level(GPIO_NUM_33, 0);
}
