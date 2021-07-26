import os
import sys

from Penalyzer.global_variables import dest_path


def main():

	try:
		text = sys.argv[1]
	except Exception as e:
		text = None

	try:
		source_style = sys.argv[2]
	except Exception as e:
		source_style = '.\\sentences'

	try:
		save_cropped_dir_path = sys.argv[3]
	except Exception as e:
		save_cropped_dir_path = None

	try:
		output_path = sys.argv[4]
	except Exception as e:
		output_path = dest_path

	print('Generating text:', text)

	source_style = ' -s ' + source_style
	dest_file = ' -d ' + output_path
	text = ' -t "' + text + '"'
	save_cropped_dir_path = ' -r ' + save_cropped_dir_path if save_cropped_dir_path is not None else ''

	if text is not None:
		os.system('python.exe -W ignore generate_single_sample.py -c ' + 
		checkpoint_path + source_style + save_cropped_dir_path + dest_file + text)


if __name__ == '__main__':
	main()