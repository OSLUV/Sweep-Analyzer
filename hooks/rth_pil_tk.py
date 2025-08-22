# hooks/rth_pil_tk.py
# Ensure Pillow's Tk integration is available before any Tk windows are created.
# This registers the 'PyImagingPhoto' command and avoids ImageTk import errors.
try:
    import PIL._tkinter_finder  # ensures Tk finds Pillow's imaging plugin
    from PIL import Image, ImageTk  # loads _imagingtk extension
except Exception as e:
    # Keep going; worst case Matplotlib falls back without an icon.
    pass
