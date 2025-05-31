"""
TELECOM CHURN ANALYSIS - MAIN RUNNER
File: main.py
Purpose: Execute analysis modules
"""

import subprocess
import sys
import argparse


def run_eda_analysis(mode='standalone'):
    """Run EDA analysis"""
    if mode == 'standalone':
        from eda import run_standalone_eda
        run_standalone_eda()
    elif mode == 'streamlit':
        subprocess.run([sys.executable, "-m", "streamlit",
                       "run", "eda.py", "streamlit"])


def run_model_training():
    """Run model training"""
    from model_training import run_standalone_model_training
    run_standalone_model_training()


def run_business_recommendations(mode='standalone'):
    """Run business recommendations"""
    if mode == 'standalone':
        from business_recommendations import run_standalone_business_analysis
        run_standalone_business_analysis()
    elif mode == 'streamlit':
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                       "business_recommendations.py", "streamlit"])


def show_menu():
    """Display menu"""
    print("\n1. Complete Analysis")
    print("2. EDA (Standalone)")
    print("3. EDA (Streamlit)")
    print("4. Model Training")
    print("5. Business Recommendations (Standalone)")
    print("6. Business Recommendations (Streamlit)")
    print("7. Exit")


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['standalone', 'streamlit'])
    parser.add_argument(
        '--module', choices=['eda', 'model', 'business', 'all'])
    args = parser.parse_args()

    # Command line execution
    if args.mode and args.module:
        if args.module == 'all':
            run_eda_analysis('standalone')
            run_model_training()
            run_business_recommendations('standalone')
        elif args.module == 'eda':
            run_eda_analysis(args.mode)
        elif args.module == 'model':
            run_model_training()
        elif args.module == 'business':
            run_business_recommendations(args.mode)
        return

    # Interactive menu
    while True:
        show_menu()
        choice = input("\nChoice: ").strip()

        if choice == "1":
            run_eda_analysis('standalone')
            run_model_training()
            run_business_recommendations('standalone')
        elif choice == "2":
            run_eda_analysis('standalone')
        elif choice == "3":
            run_eda_analysis('streamlit')
        elif choice == "4":
            run_model_training()
        elif choice == "5":
            run_business_recommendations('standalone')
        elif choice == "6":
            run_business_recommendations('streamlit')
        elif choice == "7":
            break


if __name__ == "__main__":
    main()
