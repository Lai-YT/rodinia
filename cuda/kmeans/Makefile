include ../common.mk

EXE  = kmeans_cuda kmeans_cuda_texture
OBJS = cluster.o kmeans.o kmeans_clustering.o rmse.o

.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)

.PHONY: clean
clean:
	$(RM) $(EXE) $(OBJS) output.txt
