from court import Judge
from utils import launch


def main():
    launch()
    message = [{"role": "user", "content": "How many r's are in 'Erdbeere' and 'Strawberry with cream' ?"}]

    judge = Judge()
    reply = judge.judge(messages=message)
    print(reply)


if __name__ == "__main__":
    main()
