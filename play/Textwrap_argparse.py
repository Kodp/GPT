"""
用于将一个段落格式化为指定宽度。不适合整篇文章，因为没有保留每一个段落的单独行。
"""
from tabnanny import verbose
import textwrap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('text_file', help='Input text file path', type=str)
parser.add_argument('-w', '--linewidth', help='Line width', type=int, default=40)
parser.add_argument('-o', '--output_file', help='Output target file', default=None)
#@ -v  -vv
parser.add_argument('-v', '--verbosity', action='count', help='increase output verbosity', default=0)  
parser.add_argument('-e', '--encoding', help='The text file encoding', default='utf-8')

args = parser.parse_args()

text = ""
with open(args.text_file, 'r', encoding=args.encoding) as f:
  text = f.read()
  textwraped = textwrap.fill(text, width=args.linewidth)
  if args.output_file:
    with open(args.output_file, 'w', encoding='utf-8') as fw:
      fw.write(textwraped)
    if args.verbosity >= 2:
      print(textwraped)
    elif args.verbosity >= 1:
      if len(text) > 1000:
        print(textwraped[:5*args.linewidth], end="...")
        print("\n...")
        print(textwraped[-5*args.linewidth:])
      
  else:
    print(textwraped)
  
    
