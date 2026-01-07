import csv
__all__ = ['CSVLogger', 'byte_transform', 'convert_to_latex_format','csv_to_latex']
class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def byte_transform(bytes, to, bsize=1024):
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize

    return round(r,2)

def convert_to_latex_format() -> str:
    # 탭(\t)으로 분리
    line = input()
    values = line.strip().split('\t')

    formatted_values = []
    for idx, val in enumerate(values):
        if idx % 3 == 2: continue
        # val 예: "32.33 (10.94)"
        mean_part, std_part = val.split(' (')
        std_part = std_part.rstrip(')')  # 끝의 ')' 제거

        # 소수점 첫째 자리까지만 표현 (둘째 자리에서 반올림)
        mean_float = round(float(mean_part), 1)
        std_float = round(float(std_part), 1)

        # LaTeX 형식: mean \tiny{(std)}
        formatted_values.append(f"{mean_float:.1f} \\tiny{{({std_float:.1f})}}")

    result = " & ".join(formatted_values)
    print(result)

def csv_to_latex():
    # Input data
    # numbers = input().split()
    numbers = list(map(float, input().split()))
    # numbers = [
    #     0.2703, 0.1796, 0.1988, 0.5428, 0.4525, 0.3602, 0.3741, 0.2654, 0.3154, 0.5339,
    #     0.5941, 0.2459, 0.1082, 0.0614, 0.0690, 0.4986, 0.3729, 0.3246, 0.1795, 0.1126,
    #     0.1300, 0.5014, 0.4201, 0.3487, 0.1224, 0.0844, 0.0979, 0.5078, 0.4576, 0.1990
    # ]

    # Filter out every 6th, 12th, 18th, 24th, and 30th numbers (1-based index)
    filtered_numbers = [numbers[i] for i in range(len(numbers)) if (i + 1) % 3 != 0]

    # Multiply by 100 and format to two decimal places
    transformed_numbers = [f"+{num:.1f}" for num in filtered_numbers]

    # Join the numbers with " & "
    result = " & ".join(transformed_numbers)
    print(result)

if __name__ == "__main__":
    convert_to_latex_format()
    # csv_to_latex()