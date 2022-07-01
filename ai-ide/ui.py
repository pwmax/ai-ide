import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

class UI:
    def __init__(self):
        self.window = tk.Tk()
        self.txt_edit1 = tk.Text(self.window, background='white', font=30, fg='black')
        self.txt_edit2 = tk.Text(self.window, background='white', font=30, fg='black')
        
        self.frm_buttons = tk.Frame(self.window, relief=tk.FLAT, bd=1)
       
        self.btn_open = tk.Button(self.frm_buttons, text='Open', command=self._open_file)
        self.btn_save = tk.Button(self.frm_buttons, text='Save', command=self._save_file)
        self.btn_predict = tk.Button(self.frm_buttons, text='Predict', command=self._press_predict)
        self.btn_predict_status = False
        self.txt_edit1_file = ''
        self.init()
        
    def init(self):
        # Init window
        self.window.title('')
        self.window.rowconfigure(0, minsize=800)
        self.window.columnconfigure(1, minsize=800)

        # Init buttons
        self.btn_open.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        self.btn_save.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        self.btn_predict.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        self.frm_buttons.grid(row=0, column=0, sticky='ns')

        # Init txt 
        self.txt_edit1.grid(row=0, column=1, sticky='nsew')
        self.txt_edit2.grid(row=0, column=2, sticky='nsew', padx=1)

    def update(self):
        self.window.update_idletasks()
        self.window.update()

    def get_txt1(self):
        return self.txt_edit1.get('1.0', tk.END)
    
    def get_txt2(self):
        return self.txt_edit2.get('1.0', tk.END)

    def set_txt1(self, data):
        self.txt_edit1.delete('1.0', tk.END)
        for i in data:
            self.txt_edit1.insert(tk.END, i)
    
    def set_txt2(self, data):
        self.txt_edit2.delete('1.0', tk.END)
        for i in data:
            self.txt_edit2.insert(tk.END, i)

    def _press_button(self, current_status):
        if current_status == False:
            return True
        else:
            return False

    def _press_predict(self):
        self.btn_predict_status = self._press_button(self.btn_predict_status)

    def _open_file(self):
        filepath = askopenfilename(
            filetypes=[('Python Files', '*.py')]
        )
        if not filepath:
            return
        
        self.txt_edit1_file = filepath
        self.txt_edit1.delete('1.0', tk.END)
        with open(filepath, mode='r', encoding='utf-8') as input_file:
            text = input_file.read()
            self.txt_edit1.insert(tk.END, text)
        self.window.title(f' {filepath}')

    def _save_file(self):
        if self.txt_edit1_file == '':
            return
        
        with open(self.txt_edit1_file, mode='w', encoding='utf-8') as save_file:
            data = self.get_txt1()
            for i in data:
                save_file.write(i)
    
    def _predict_file(self):
        if self.btn_predict_status == True:
            self.btn_predict_status = False
        
        elif self.btn_predict_status == False:
            self.btn_predict_status = True

if __name__ == '__main__':
    ui = UI()
    while 1:
        ui.update()
      