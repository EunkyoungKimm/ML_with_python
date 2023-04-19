gun = 10

def checkpoint_return(gun,soldiers):
    gun = gun - soldiers
    print(f"[함수내]남은 총: {gun}")
    return gun

print(f'전체총:{gun}')

gun = checkpoint_return(gun,2)
print(gun)
