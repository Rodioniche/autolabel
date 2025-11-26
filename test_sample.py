"""
Утилита для ПК: отправляет камере OpenMV команду сделать снимок и сохраняет
возвращённый JPEG на диск. Перед запуском:

1. Подключите камеру по USB и запустите в OpenMV IDE скрипт `openmv_capture_slave.py`.
2. Узнайте имя COM-порта (в Диспетчере устройств Windows он отображается как
   «OpenMV Virtual COM Port»).
3. Установите зависимость: `pip install pyserial`.

Пример запуска:
    python test_sample.py --port COM7 --output-folder captures
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import struct
import sys

import serial

CMD_PING = b"\x70"
CMD_CAPTURE = b"\x71"
PING_REPLY = b"OK"


def read_exact(port: serial.Serial, size: int, timeout_message: str) -> bytes:
    """Читает ровно size байт или выбрасывает TimeoutError."""
    buffer = bytearray()
    while len(buffer) < size:
        chunk = port.read(size - len(buffer))
        if not chunk:
            raise TimeoutError(timeout_message)
        buffer.extend(chunk)
    return bytes(buffer)


def capture_once(port_name: str, output_path: Path, baudrate: int, timeout: float) -> Path:
    with serial.Serial(port=port_name, baudrate=baudrate, timeout=timeout) as ser:
        ser.reset_input_buffer()
        ser.write(CMD_PING)
        reply = read_exact(ser, len(PING_REPLY), "Не получил ping-ответ от камеры.")
        if reply != PING_REPLY:
            raise RuntimeError(
                f"Камера не отвечает на ping (ожидалось {PING_REPLY!r}, получено {reply!r}). "
                "Проверьте, что на OpenMV запущен скрипт openmv_capture_slave.py."
            )

        ser.write(CMD_CAPTURE)
        raw_len = read_exact(ser, 4, "Не дождался длины JPEG от камеры.")
        (jpeg_len,) = struct.unpack("<I", raw_len)
        data = read_exact(ser, jpeg_len, "Не удалось дочитать JPEG полностью.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path


def default_filename(folder: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder / f"openmv_{timestamp}.jpg"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сохранить фото с OpenMV Cam на ПК.")
    parser.add_argument("--port", required=True, help="COM-порт камеры (например, COM7).")
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Скорость порта. 115200 подходит по умолчанию.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Таймаут ожидания данных в секундах.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("captures"),
        help="Каталог, куда складывать снимки.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Полный путь к файлу. Если не указан, будет создан по дате.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    target = args.output or default_filename(args.output_folder)
    try:
        saved = capture_once(args.port, target, args.baudrate, args.timeout)
    except (RuntimeError, TimeoutError, serial.SerialException) as exc:
        sys.exit(f"Ошибка: {exc}")
    print(f"Снимок сохранён: {saved}")


if __name__ == "__main__":
    main()

