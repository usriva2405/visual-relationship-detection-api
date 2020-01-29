# config_loader.py
import yaml
import os
import re

from dotenv import load_dotenv
from pathlib import Path
import logging
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Interact with configuration variables."""

    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)

    __configFilePath = (os.path.join(os.getcwd(), 'conf/config.yaml'))
    __configParser = yaml.load(open(__configFilePath), Loader=yaml.SafeLoader)

    @classmethod
    def __getenv(cls):
        """DEV or PROD"""
        # env = cls.env_config['global']['env']
        env = cls.__getenvvar("env")
        if env is '' or env is None:
            # use default value as DEV, in case env is not set
            env = 'DEV'
        return env

    @classmethod
    def __getenvvar(cls, key):
        value = os.getenv(key)
        if value is '' or value is None:
            # use default value as DEV, in case env is not set
            value = None
        return value

    # @classmethod
    # def get_config_val(cls, key, *args, **kwargs):
    #     # TODO change it to key1.key2.key3, parse the string, extract depth
    #     """Get prod values from config.yaml."""
    #     env = cls.__getenv()
    #     key_1depth = kwargs.get('key_1depth', None)
    #     key_2depth = kwargs.get('key_2depth', None)
    #     key_3depth = kwargs.get('key_3depth', None)
    #     try:
    #         if key_1depth is not None:
    #             if key_2depth is not None:
    #                 if key_3depth is not None:
    #                     return str(cls.__configParser[env][key][key_1depth][key_2depth][key_3depth])
    #                 else:
    #                     return str(cls.__configParser[env][key][key_1depth][key_2depth])
    #             else:
    #                 return str(cls.__configParser[env][key][key_1depth])
    #         else:
    #             return str(cls.__configParser[env][key])
    #     except Exception as e:
    #         print(e)
    #         print('invalid key structure passed for retrieving value from config.yaml')
    #     return None

    @classmethod
    def get_config_val(cls, key, *args, **kwargs):
        # TODO change it to key1.key2.key3, parse the string, extract depth
        """Get prod values from config.yaml."""
        env = cls.__getenv()
        config_value = None
        value = None
        key_1depth = kwargs.get('key_1depth', None)
        key_2depth = kwargs.get('key_2depth', None)
        key_3depth = kwargs.get('key_3depth', None)
        try:
            if key_1depth is not None:
                if key_2depth is not None:
                    if key_3depth is not None:
                        config_value = str(cls.__configParser[env][key][key_1depth][key_2depth][key_3depth])
                    else:
                        config_value = str(cls.__configParser[env][key][key_1depth][key_2depth])
                else:
                    config_value = str(cls.__configParser[env][key][key_1depth])
            else:
                config_value = str(cls.__configParser[env][key])
        except Exception as e:
            logger.error(e)
            logger.error('invalid key structure passed for retrieving value from config.yaml')
            config_value = None

        value = None

        try:
            if config_value is not None:
                # check if value is an environment variable reference
                m = re.search(r'^\${([A-Za-z_-]+)}', config_value)
                if m:
                    value = cls.__getenvvar(m.group(1))
                else:
                    value = config_value
            else:
                value = None
        except Exception as e:
            logger.error(e)
            value = None

        logger.info("config value for key {0} : {1}".format(key, value))

        return value
