#include <stdio.h>
#include "driver/gpio.h"
#include "hal/gpio_types.h"
#include "camera.h"
#include "led.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void) {
    init_led();
    init_camera();
    while (true) {
        /* code */
    }
}
