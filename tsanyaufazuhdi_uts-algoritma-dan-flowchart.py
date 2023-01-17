i = 0
nilai = [60,70,90,65,75,55,70,55,85,990,660]
for i in range(11):
    print("Nilai ", nilai[i])

jumlah = nilai[0] +nilai[1] +nilai[2] +nilai[3] +nilai[4] +nilai[5] +nilai[6] +nilai[7] +nilai[8] +nilai[9] +nilai[10]
print("Jumlah nilai = ", jumlah)
rata = jumlah/11
print("Rata-rata Nilai = {:.2f}".format(rata))

for i in range (11):
    deviasi = 1/(11-1) * (nilai[i] - rata)**2
    print("Deviasi = {:.2f}".format(deviasi))

