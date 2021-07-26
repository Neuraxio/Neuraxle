import argparse

import urllib3

# Here is the command I use to verify all libraries installed with pip.
# python license_checker_v2.py --dependencies $(cut -d '=' -f 1 <<< $(pip freeze))
# Alternatively, if you are processing directly a requirements.txt file, you can use the following command to parse
# awk -F'[>=<]' '{print $1}' requirements.txt

parser = argparse.ArgumentParser()
parser.add_argument('--dependencies', nargs='+', required=True,
                    help="A list of python library name you want to check the license of.")
parser.add_argument('--accepted_licenses', nargs='*',
                    help="A list of license which are considered acceptable for your project.",
                    default=["Apache Software License", "Apache 2.0", "BSD", "ZLIB", "MIT", "Unlicense", "CC0", "CC-BY","PSF", "MPL", "Mozilla Public License 2.0", "Historical Permission Notice and Disclaimer", "HPND"])
parser.add_argument('--forbidden_licenses', nargs='*',
                    help="A list of license which are considered problematic for your project.",
                    default=["GNU", "GPL", "Commons Clause", "BY-N"])
args = parser.parse_args()

python_dependencies = args.dependencies
pypi_pages = {}

for library_name in python_dependencies:
    url = f"https://pypi.org/project/{library_name}/"
    http_pool = urllib3.connection_from_url(url)
    result = http_pool.urlopen('GET', url)
    html_page = result.data.decode('utf-8')
    pypi_pages[library_name] = html_page


def parse_html(html_page):
    lines = html_page.split('\n')

    for i, l in enumerate(lines):
        if ("<a href=" in l) and ("License:" in l):
            return lines[i + 1].split('::')[1].strip('\n')

    for i, l in enumerate(lines):
        if "<strong>License:</strong> " in l:
            return lines[i].replace("<p><strong>License:</strong> ", "").replace('</p>', '')

    raise ValueError("Unable to find license in html page")


unknown_licenses = []
library_license_dict = {}
accepted_libraries = []
refused_libraries = []
maybe_libraries = []


def is_license_in_list(license, license_list):
    for l in license_list:
        if l.lower() in license.lower():
            return True
    return False


for library_name in python_dependencies:
    try:
        library_license = parse_html(pypi_pages[library_name])
        library_license_dict[library_name] = library_license
        print(f"{library_name}: {library_license}")
        # First checks if its refused_licenses, then if its in accepted_licenses, else add in the maybe list
        # TODO : Should use regex instead?

        if is_license_in_list(library_license, args.forbidden_licenses):
            refused_libraries.append(library_name)
        elif is_license_in_list(library_license, args.accepted_licenses):
            accepted_libraries.append(library_name)
        else:
            maybe_libraries.append(library_name)

    except Exception as e:
        print(f"{library_name}: {e}")
        unknown_licenses.append(library_name)


def plurial(lst, _if='s', _else=''):
    return _if if len(lst) > 1 else _else

if len(unknown_licenses) > 0:
    print(f"Couldn't find the license{plurial(unknown_licenses)} of the following dependencies: {unknown_licenses}")

print(f"\nThe following dependenc{plurial(accepted_libraries, 'y', 'ies')} have an accepted license: {accepted_libraries}")

if len(refused_libraries) > 0:
    print(f"The following dependencie{plurial(refused_libraries, 'y', 'ies')} have forbidden license(s):")
    for library_name in refused_libraries:
        print(f"  {library_name}: {library_license_dict[library_name]}")

if len(maybe_libraries) > 0:
    print(f"The following dependencie{plurial(maybe_libraries, 'y', 'ies')} have license which needs to be reviewed: ")
    for library_name in maybe_libraries:
        print(f"  {library_name}: {library_license_dict[library_name]}")


assert len(refused_libraries) == 0 and len(maybe_libraries) == 0 and len(unknown_licenses) == 0
