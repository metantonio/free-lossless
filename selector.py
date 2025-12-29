import win32gui
import win32process
import psutil

class WindowSelector:
    @staticmethod
    def get_visible_windows():
        """
        Returns a list of visible windows with their titles and HWNDs.
        """
        def enum_handler(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    # Get process name
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        process = psutil.Process(pid)
                        proc_name = process.name()
                        windows.append({"hwnd": hwnd, "title": title, "process": proc_name})
                    except:
                        pass
        
        windows = []
        win32gui.EnumWindows(enum_handler, windows)
        return windows

    @staticmethod
    def get_window_rect(hwnd):
        """
        Returns the (left, top, right, bottom) of the window.
        """
        return win32gui.GetWindowRect(hwnd)

if __name__ == "__main__":
    selector = WindowSelector()
    wins = selector.get_visible_windows()
    print("Visible Windows:")
    for i, w in enumerate(wins):
        print(f"[{i}] {w['title']} ({w['process']})")
