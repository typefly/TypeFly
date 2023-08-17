#include <stdio.h>
#include "driver/gpio.h"
#include "hal/gpio_types.h"
#include "camera.h"
#include "led.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_camera.h"

static const char *TAG = "main";

void app_main(void) {
    init_led();
    init_camera();
    
    while (true) {
        ESP_LOGI(TAG, "Taking picture...");
        camera_fb_t *pic = esp_camera_fb_get();

        // use pic->buf to access the image
        ESP_LOGI(TAG, "Picture taken! Its size was: %zu bytes", pic->len);
        esp_camera_fb_return(pic);

        vTaskDelay(5000 / portTICK_PERIOD_MS);
    }
}
