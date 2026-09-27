import logging
import unittest

from ballontranslator.utils.logger import (
    ColoredFormatter,
    ColoredLogger,
    FORMAT,
    START_LEVEL,
    STOP_LEVEL,
    logger,
)


class TestLoggerFormatting(unittest.TestCase):

    def setUp(self):
        self.formatter = ColoredFormatter(FORMAT, use_color=True)

    def test_no_extra_whitespace_in_levelname(self):
        """Level names must not contain fixed-width padding whitespace like [INFO   ]."""
        for level, expected in [
            (logging.INFO, "INFO"),
            (logging.DEBUG, "DEBUG"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (START_LEVEL, "START"),
            (STOP_LEVEL, "STOP"),
        ]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            self.formatter.format(record)
            # The raw text inside ANSI escape codes must not have trailing whitespace
            self.assertIn(f"{expected}\x1b[0m", record.levelname2)
            self.assertNotIn("   ", record.levelname2)

    def test_logger_start_and_stop_methods(self):
        """ColoredLogger must have .start() and .stop() working properly."""
        test_logger = ColoredLogger("test_start_stop")
        test_logger.setLevel(logging.DEBUG)

        records = []
        class TestHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = TestHandler()
        test_logger.addHandler(handler)

        test_logger.start("Process started")
        test_logger.stop("Process finished")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].levelno, START_LEVEL)
        self.assertEqual(records[0].levelname, "START")
        self.assertEqual(records[0].getMessage(), "Process started")

        self.assertEqual(records[1].levelno, STOP_LEVEL)
        self.assertEqual(records[1].levelname, "STOP")
        self.assertEqual(records[1].getMessage(), "Process finished")


if __name__ == '__main__':
    unittest.main()
