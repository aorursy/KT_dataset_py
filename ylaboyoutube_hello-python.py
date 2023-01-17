# 変数を定義

spam_amount = 0

print(spam_amount)



# =は等号ではなく代入

# spam_amount = spam_amount + 4

spam_amount += 4

print(spam_amount)



# 条件分岐

# Pythonではインデントが大切

# if の条件の後には:を忘れずに

if spam_amount > 0:

  print("But I don't want ANY spam!")



viking_song = "Spam " * spam_amount

print(viking_song)
spam_amount = 0
print(spam_amount)
spam_amount = spam_amount + 4
if spam_amount > 0:

    print("But I don't want ANY spam!")



viking_song = "Spam Spam Spam"

print(viking_song)
viking_song
"But I don't want ANY spam!"
viking_song = "Spam " * spam_amount

print(viking_song)
spam_amount = 0
type(spam_amount)
type(19.95)
type("spam")
# a + b	足し算

# a - b	引き算

# a * b	掛け算

# a / b	割り算

# a // b	割り算の商

# a % b	割り算のあまり

# a ** b	aのb乗

# -a 負
print(5 / 2)

print(6 / 2)
print(5 // 2)

print(6 // 2)
8 - 3 + 2
-3 + 4 * 2
hat_height_cm = 25

my_height_cm = 190

# 帽子を合わせた身長は何メートルでしょうか？

total_height_meters = hat_height_cm + my_height_cm / 100

print("Height in meters =", total_height_meters, "?")
#先に計算したい場合は()で囲みましょう

total_height_meters = (hat_height_cm + my_height_cm) / 100

print("Height in meters =", total_height_meters)
#最大と最小

print(min(1, 2, 3))

print(max(1, 2, 3))
#絶対値

print(abs(32))

print(abs(-32))
#型変換

print(float(10))

print(int(3.33))

print(int('807') + 1)