from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['configs/settings.yaml',
                    'configs/.secrets.yaml'],
    environments=True,
    load_dotenv=True,
    env_switcher="DYNACONF_ENV",
    env='development',
)
