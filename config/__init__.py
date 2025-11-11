import yaml

with open('./config/config.yml', 'r') as f:
    config = yaml.safe_load(f)

locals().update(config)
