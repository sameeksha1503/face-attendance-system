from src import capture, train

def main_menu():
    while True:
        print("\n"+"-"*30)
        print(" Face Recongnization Attendence System ")
        print("-"*30)
        print("1. ADD New Student")
        print("2. Train Model")
        print("2. Take live attendene")
        print("4. Exit")
        print("-"*30)

        choice= input("Selection option 1-4 :").strip()

        if choice=="1":
            capture.add_new_student()
        elif choice=="2":
            train.train_model()
        elif choice=="3":
            print("take attendence")
        elif choice=="4":
            print("exit")
            break
        else:
            print("invalid choce")
            
if __name__=="__main__":
     main_menu()