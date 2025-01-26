from app.gui import setup_gui
from app.video_processing import start_video_processing

def main():
    # Setup GUI
    setup_gui()

    # Start real-time video processing
    start_video_processing()

if __name__ == "__main__":
    main()
