# Clear unfinished targets
.DELETE_ON_ERROR:

# Define phony targets
.PHONY: clean

#----------------------------------------------------------
# Main targets
#----------------------------------------------------------

all: spline_eval_funcs$(SO_EXT) cubic_uniform_spline_eval_funcs$(SO_EXT) #splines$(SO_EXT)

.ACC.$(ACC):
	@rm -f .ACC.*
	@touch $@

spline_eval_funcs$(SO_EXT): $(NAME_PREFIX)spline_eval_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif

cubic_uniform_spline_eval_funcs$(SO_EXT): $(NAME_PREFIX)cubic_uniform_spline_eval_funcs.py .ACC.$(ACC)
ifneq ($(ACC), pythran)
	$(TOOL) $< $(TOOL_FLAGS)
else
	$(TOOL) $< $(TOOL_FLAGS) -o $@
endif

#splines$(SO_EXT): $(NAME_PREFIX)splines.py .ACC.$(ACC)
#ifneq ($(ACC), pythran)
#	$(TOOL) $< $(TOOL_FLAGS)
#else
#	$(TOOL) $< $(TOOL_FLAGS) -o $@
#endif

clean:
	rm -f *.o *.so *.mod .ACC.pycc .lock_acquisition.lock
	pyccel-clean

