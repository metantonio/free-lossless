import tkinter as tk
from tkinter import ttk
from selector import WindowSelector

class GameSelectorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lossless Frame Gen - Select Game")
        self.root.geometry("500x400")
        
        self.selected_window = None
        self.selector = WindowSelector()
        
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        label = tk.Label(self.root, text="Select the game window to apply Frame Generation:", pady=10)
        label.pack()

        # Listbox
        self.tree = ttk.Treeview(self.root, columns=("Title", "Process"), show="headings")
        self.tree.heading("Title", text="Window Title")
        self.tree.heading("Process", text="Process Name")
        self.tree.column("Title", width=300)
        self.tree.column("Process", width=150)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        mode_frame = tk.Frame(self.root, pady=5)
        mode_frame.pack()
        tk.Label(mode_frame, text="Capture Mode: ").grid(row=0, column=0)
        self.mode_var = tk.StringVar(value="dxcam")
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, values=["dxcam", "bitblt"], state="readonly", width=10)
        self.mode_combo.grid(row=0, column=1)

        # FPS Selection
        fps_frame = tk.Frame(self.root, pady=5)
        fps_frame.pack()
        tk.Label(fps_frame, text="Target FPS: ").grid(row=0, column=0)
        self.fps_var = tk.IntVar(value=60)
        self.fps_scale = tk.Scale(fps_frame, from_=30, to=120, orient=tk.HORIZONTAL, variable=self.fps_var, resolution=1, length=200)
        self.fps_scale.grid(row=0, column=1)

        # Buttons
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack()
        
        refresh_btn = tk.Button(btn_frame, text="Refresh List", command=self._refresh_list)
        refresh_btn.grid(row=0, column=0, padx=5)
        
        select_btn = tk.Button(btn_frame, text="Start Frame Gen", command=self._on_select, bg="#4CAF50", fg="white")
        select_btn.grid(row=0, column=1, padx=5)

    def _refresh_list(self):
        # Clear
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        windows = self.selector.get_visible_windows()
        for w in windows:
            self.tree.insert("", tk.END, values=(w["title"], w["process"]), iid=str(w["hwnd"]))

    def _on_select(self):
        selected = self.tree.selection()
        if selected:
            hwnd = int(selected[0])
            title = self.tree.item(selected[0], "values")[0]
            self.selected_window = {
                "hwnd": hwnd, 
                "title": title,
                "mode": self.mode_var.get(),
                "fps": self.fps_var.get()
            }
            self.root.destroy()

    def get_selection(self):
        self.root.mainloop()
        return self.selected_window

if __name__ == "__main__":
    ui = GameSelectorUI()
    selection = ui.get_selection()
    print(f"User selected: {selection}")
