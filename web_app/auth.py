from functools import wraps
from flask import redirect, url_for, session
from authlib.integrations.flask_client import OAuth


class OAuthSSO:
    def __init__(
        self,
        app,
        client_name="keycloak",
        client_id=None,
        client_secret=None,
        server_metadata_url=None,
        scope="openid profile email",
    ):
        self.app = app
        self.client_name = client_name
        self.oauth = OAuth(app)
        self.client = self.oauth.register(
            name=client_name,
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=server_metadata_url,
            client_kwargs={"scope": scope},
        )

    def login(self):
        redirect_uri = url_for("auth_callback", _external=True)
        return self.client.authorize_redirect(redirect_uri)

    def callback(self):
        token = self.client.authorize_access_token()
        session["user"] = token.get("userinfo") or token.get("id_token_claims")
        session["id_token"] = token.get("id_token")
        return redirect(url_for("index"))

    def logout(self):
        id_token = session.pop("id_token", None)
        session.pop("user", None)
        session.pop("token", None)

        end_session_url = self.client.server_metadata.get("end_session_endpoint")
        if end_session_url and id_token:
            redirect_uri = url_for("index", _external=True)
            return redirect(
                f"{end_session_url}?id_token_hint={id_token}&post_logout_redirect_uri={redirect_uri}"
            )

        return redirect(url_for("index"))

    def logged_in(self):
        return session.get("user") is not None
