import sys
from Performance import performance_test
from Time_and_accuracy import time_and_accuracy_test


def main():
    while True:
        print("\n" + "=" * 55)
        print("          BENCHMARK TESTING MENU")
        print("=" * 55)
        print("1. Run Performance Test (Continuous Benchmark)")
        print("2. Run Time and Accuracy Test (Scalability)")
        print("3. Run All Tests")
        print("4. Exit")
        print("=" * 55)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            print("\n[*] Starting Performance Test...")
            performance_test()
        elif choice == '2':
            print("\n[*] Starting Time and Accuracy Test...")
            time_and_accuracy_test()
        elif choice == '3':
            print("\n[*] Starting Performance Test...")
            performance_test()
            print("\n[*] Starting Time and Accuracy Test...")
            time_and_accuracy_test()
        elif choice == '4':
            print("\nExiting program. Goodbye!")
            sys.exit(0)
        else:
            print("\n[!] Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()