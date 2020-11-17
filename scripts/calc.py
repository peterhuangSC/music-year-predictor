
denominator_file = open('./sum_den.txt')
denominator_str= denominator_file.readline()[:-1]

denominator = (eval(denominator_str))
print('Sum of all denominators: {}'.format(denominator))

numerator_file = open('./sum.txt')
numerator_str = numerator_file.readline()[:-1]

numerator = (eval(numerator_str))
print('Sum of all numerators: {}'.format(numerator))
print('Final Answer: {} average'.format(numerator / denominator))

