import uvicorn

if __name__ == "__main__":
    print("Starting Australian Financial Planning AI Web Interface...")
    print("Access the interface at http://localhost:8000/")
    uvicorn.run("app.web.app:app", host="0.0.0.0", port=8000, reload=True) 