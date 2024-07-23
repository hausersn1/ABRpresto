from ABRpresto import XCsub
import os


def main_process():
    import argparse
    parser = argparse.ArgumentParser('auto-th')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-r', '--recursive', action='store_true')
    args = parser.parse_args()

    if args.recursive:
        for pth in args.paths:
            filenames = [filename for filename in os.listdir(pth) if filename.endswith('.csv')]
            print(f'Found {len(filenames)} .csv files in {pth}, running XCsub on each:')
            for filename in filenames:
                XCsub.run_fit(os.path.join(pth, filename))
    else:
        for pth in args.paths:
            if os.path.isdir(pth):
                raise RuntimeError(f'{pth} is a directory, pass only full paths to csv files.')
            XCsub.run_fit(pth)

if __name__ == '__main__':
    main_process()
