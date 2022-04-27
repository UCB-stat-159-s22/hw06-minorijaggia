.PHONY: env
env: 
	bash -i envsetup.sh

.PHONY: html
html:
	jupyterbook build .

.PHONY: html-hub
html-hub:
	jupyter-book config sphinx .
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}/proxy/absolute/8000
	cd _build/html 
	python -m http.server

.PHONY: clean
clean:
	rm figures/*.png
	rm data/*.csv
	rm audio/*.wav
	rm -rf build
	rm -rf ligotools.egg-info
	rm -rf _build