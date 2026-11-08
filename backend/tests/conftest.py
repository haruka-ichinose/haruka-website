import os
import sys
from pathlib import Path

# Required settings with no defaults; set before app modules import config.
for key in ("GITHUB_TOKEN", "SMTP_USER", "SMTP_PASSWORD", "CONTACT_RECIPIENT"):
    os.environ.setdefault(key, "test")

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
