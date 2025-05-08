import os
import warnings
from app.controller import MainWindowController

warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")

os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def main():
    try:
        controller = MainWindowController()
        controller.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
