
all: build

source:
	. /Users/guojun/esp/esp-idf/export.sh

build:
	idf.py build

flash:
	@usbport=$$(ls /dev/cu.usbserial-* | head -1); \
	if [ -z "$$usbport" ]; then \
		echo "No USB port found"; \
	else \
		idf.py -p $$usbport flash; \
	fi

.PHONY: all build flash