from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    disabled: bool = False

async def get_current_active_user(token: str = Depends(oauth2_scheme)):
    user = User(username=token)
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

def setup_auth_routes(app: FastAPI):
    @app.post("/token")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        # Basic auth - replace with proper authentication in production
        return {"access_token": form_data.username, "token_type": "bearer"} 