# scripts/file_watcher.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from scripts.content_manager import gather_all_content
from scripts.ai_engine import create_or_update_faiss_db
from threading import Timer

WATCH_FOLDER = "project_files/"  # your file directory
debounce_timer = None

def schedule_update():
    global debounce_timer
    if debounce_timer:
        debounce_timer.cancel()
    debounce_timer = Timer(2.0, run_update)  # wait 2 seconds before running
    debounce_timer.start()

def run_update():
    aggregated = gather_all_content()
    create_or_update_faiss_db(aggregated)
    print("[Watcher] FAISS + DB updated.")

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f"[Watcher] Detected change in {event.src_path}")
            aggregated = gather_all_content()
            create_or_update_faiss_db(aggregated)
            print("[Watcher] FAISS + DB updated.")

    def on_created(self, event):
        if not event.is_directory:
            print(f"[Watcher] Detected new file {event.src_path}")
            aggregated = gather_all_content()
            create_or_update_faiss_db(aggregated)
            print("[Watcher] FAISS + DB updated.")

def start_watcher():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=True)
    observer.start()
    print(f"[Watcher] Monitoring folder: {WATCH_FOLDER}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
