!pip install pyfiglet
from kaggle_secrets import UserSecretsClient



secret_label = "my_cool_word_art"

secret_value = UserSecretsClient().get_secret(secret_label)





import pyfiglet



ascii_banner = pyfiglet.figlet_format(secret_value)

print(ascii_banner)