idf_component_register(
    SRCS
        "main.c"
        "camera.c"
        "led.c"
    INCLUDE_DIRS
        "."
        "include"
    REQUIRES "../esp32-camera/"
)