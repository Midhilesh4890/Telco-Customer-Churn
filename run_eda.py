# =============================================================================
# EDA RUNNER SCRIPT
# File: run_eda.py
# Purpose: Easy execution of both standalone and Streamlit EDA versions
# =============================================================================

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'plotly', 'streamlit'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def run_standalone_eda():
    """Run standalone EDA version"""
    print("Running Standalone EDA Analysis...")
    print("=" * 40)

    try:
        from eda import run_standalone_eda
        analyzer, charts = run_standalone_eda()

        if analyzer:
            print("\nStandalone EDA completed successfully!")
            print(f"Charts saved in: {analyzer.charts_dir}/")
            print("\nGenerated files:")

            chart_files = list(Path(analyzer.charts_dir).glob("*.html"))
            for file in chart_files:
                print(f"  - {file.name}")

            return True
        else:
            print("Error: Failed to run standalone EDA")
            return False

    except Exception as e:
        print(f"Error running standalone EDA: {str(e)}")
        return False


def run_streamlit_eda():
    """Run Streamlit EDA version"""
    print("Launching Streamlit EDA Application...")
    print("=" * 40)

    try:
        # Check if interactive_eda.py exists
        if not Path("interactive_eda.py").exists():
            print("Error: interactive_eda.py not found in current directory")
            return False

        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "interactive_eda.py", "streamlit"
        ])

        return True

    except Exception as e:
        print(f"Error launching Streamlit: {str(e)}")
        return False


def main():
    """Main execution function"""
    print("TELECOM CHURN EDA - EXECUTION OPTIONS")
    print("=" * 45)

    # Check requirements
    if not check_requirements():
        return

    print("\nAvailable options:")
    print("1. Run Standalone EDA (generates HTML files)")
    print("2. Run Streamlit EDA (interactive web app)")
    print("3. Run both versions")
    print("4. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                success = run_standalone_eda()
                if success:
                    print("\nStandalone EDA completed!")
                    input("Press Enter to continue...")
                break

            elif choice == "2":
                print("\nLaunching Streamlit...")
                print("The application will open in your default browser.")
                print("Press Ctrl+C to stop the Streamlit server.")
                run_streamlit_eda()
                break

            elif choice == "3":
                print("Running standalone EDA first...")
                success = run_standalone_eda()

                if success:
                    print("\nNow launching Streamlit...")
                    input("Press Enter to continue to Streamlit...")
                    run_streamlit_eda()
                break

            elif choice == "4":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
