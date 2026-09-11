## Keycloak Setup
0. **Create keycloak instance**
```docker
docker run -p 8080:8080 -e KEYCLOAK_ADMIN=admin -e KEYCLOAK_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:21.1.1 start-dev
```
- Admin console at http://localhost:8080/
- Username and password: `admin`

1. **Create a Realm**  
   - Go to Keycloak Admin → Realms → Add Realm

2. **Create a Client**  
   - Go to Clients → Create → Client ID: `flask-app` → Client Protocol: `openid-connect`  
   - **Settings**:  
     - Client authentication: `on`
     - Root URL: url to app (e.g. `http://127.0.0.1:9000`)
     - Valid Redirect URIs: `/auth`  
     - Web Origins: `*`  
     - Valid Post Logout Redirect URIs: `/`

3. **Get Client Secret**  
   - Go to **Credentials** → copy the secret  

4. **Create Test User**  
   - Users → Add User → username/email → set password

## Flask Setup
1. **Install dependencies**
- `pip install authlib requests`

2. **Edit OAuthSSO registration**
```py
sso = OAuthSSO(
    app,
    client_name="keycloak",
    client_id="flask-app", # Copy from Keycloak
    client_secret="YOUR_SECRET_HERE", # Copy from Keycloak
    server_metadata_url="{keycloak_path}/realms/dev/.well-known/openid-configuration" # e.g (https://localhost:8080//realms/dev/.well-known/openid-configuration)
)
```