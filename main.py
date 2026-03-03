import uvicorn

from config.map_config import load_config
from server import create_app

if __name__ == "__main__":
    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=8003) 