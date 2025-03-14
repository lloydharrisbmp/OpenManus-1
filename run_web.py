import uvicorn
import webbrowser
import threading
import time

def open_browser():
    """Open web browser after a short delay to ensure server is running."""
    time.sleep(1.5)
    print("Opening web browser...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("Starting Australian Financial Planning AI Web Interface...")
    print("Access the interface at http://localhost:8000/")
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Run the server
    uvicorn.run("app.web.app:app", host="0.0.0.0", port=8000, reload=True) 