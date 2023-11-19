SHELL := /bin/bash
SOURCEDIR = source
BUILDDIR = build

all:
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going -P

clean:
	rm -rf "$(BUILDDIR)"
