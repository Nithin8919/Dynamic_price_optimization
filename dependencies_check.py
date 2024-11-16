import pkg_resources
from pkg_resources import working_set
import subprocess
import sys

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_zenml_compatibility():
    # Get ZenML's dependencies
    try:
        zenml_dist = pkg_resources.get_distribution('zenml')
        zenml_version = zenml_dist.version
        print(f"Installed ZenML version: {zenml_version}")
    except pkg_resources.DistributionNotFound:
        print("ZenML is not installed")
        return

    # Check specific packages
    packages_to_check = {
        'pandas': None,
        'numpy': None,
        'scikit-learn': None,
        'kafka-python': None,
        'matplotlib': None,
        'seaborn': None,
        'plotly': None,
        'psycopg2': None,
        'sqlalchemy': '1.3.24',  # Specified version
        'scipy': None,
        'statsmodels': None,
        'mlflow': None,
        'apache-airflow': None
    }

    print("\nChecking package versions:")
    print("-" * 50)

    conflicts = []
    for package, required_version in packages_to_check.items():
        installed_version = get_installed_version(package)
        
        if installed_version:
            print(f"{package}: {installed_version}")
            
            # Special check for SQLAlchemy
            if package == 'sqlalchemy':
                if installed_version.startswith('1.3.24'):
                    print("✓ SQLAlchemy version matches required version")
                else:
                    conflicts.append(f"SQLAlchemy version mismatch: Installed {installed_version}, required 1.3.24")
        else:
            print(f"{package}: Not installed")

    print("\nAnalyzing potential conflicts:")
    print("-" * 50)
    
    if conflicts:
        print("Found conflicts:")
        for conflict in conflicts:
            print(f"! {conflict}")
        print("\nRecommended actions:")
        print("1. Create a new virtual environment")
        print("2. Install ZenML first: pip install zenml")
        print("3. Then install other dependencies one by one")
        print("4. For SQLAlchemy specifically: pip install sqlalchemy==1.3.24")
    else:
        print("No obvious version conflicts detected")

if __name__ == "__main__":
    check_zenml_compatibility()