"use client";

import { useState, useEffect, useCallback } from "react";
import { Trash2, Pencil, Search, X, RefreshCw, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "";

interface Memory {
  id: string;
  memory: string;
  hash?: string;
  metadata?: {
    project_id?: string;
    [key: string]: unknown;
  };
  created_at?: string;
  updated_at?: string;
}

export function MemoryManager({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [editText, setEditText] = useState("");
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const fetchMemories = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/memories`);
      if (!res.ok) {
        throw new Error(`Failed to fetch memories: ${res.status}`);
      }
      const data = await res.json();
      setMemories(data.memories || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch memories");
    } finally {
      setLoading(false);
    }
  }, []);

  const searchMemories = async () => {
    if (!searchQuery.trim()) {
      fetchMemories();
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/memories/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: searchQuery }),
      });
      if (!res.ok) {
        throw new Error(`Search failed: ${res.status}`);
      }
      const data = await res.json();
      setMemories(data.memories || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const updateMemory = async () => {
    if (!editingMemory) return;
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/memories/${editingMemory.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: editText }),
      });
      if (!res.ok) {
        throw new Error(`Update failed: ${res.status}`);
      }
      setEditingMemory(null);
      fetchMemories();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Update failed");
    } finally {
      setLoading(false);
    }
  };

  const deleteMemory = async (id: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/memories/${id}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        throw new Error(`Delete failed: ${res.status}`);
      }
      setDeleteConfirm(null);
      fetchMemories();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setLoading(false);
    }
  };

  const deleteAllMemories = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/memories`, {
        method: "DELETE",
      });
      if (!res.ok) {
        throw new Error(`Delete all failed: ${res.status}`);
      }
      setDeleteConfirm(null);
      fetchMemories();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete all failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      fetchMemories();
    }
  }, [open, fetchMemories]);

  const startEdit = (memory: Memory) => {
    setEditingMemory(memory);
    setEditText(memory.memory);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Brain className="size-5" />
            Memory Manager
          </DialogTitle>
          <DialogDescription>
            View, edit, and delete memories stored by Clara
          </DialogDescription>
        </DialogHeader>

        {/* Search bar */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
            <Input
              placeholder="Search memories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && searchMemories()}
              className="pl-9"
            />
          </div>
          <Button variant="outline" size="icon" onClick={searchMemories}>
            <Search className="size-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={fetchMemories}>
            <RefreshCw className="size-4" />
          </Button>
        </div>

        {/* Error display */}
        {error && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Memory list */}
        <div className="flex-1 overflow-y-auto space-y-2 min-h-[300px]">
          {loading ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Loading...
            </div>
          ) : memories.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              No memories found
            </div>
          ) : (
            memories.map((memory) => (
              <div
                key={memory.id}
                className="group rounded-lg border bg-card p-3 hover:bg-accent/50 transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm flex-1">{memory.memory}</p>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      onClick={() => startEdit(memory)}
                    >
                      <Pencil className="size-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7 text-destructive hover:text-destructive"
                      onClick={() => setDeleteConfirm(memory.id)}
                    >
                      <Trash2 className="size-3" />
                    </Button>
                  </div>
                </div>
                {memory.metadata?.project_id && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Project: {memory.metadata.project_id}
                  </p>
                )}
              </div>
            ))
          )}
        </div>

        {/* Footer with count and delete all */}
        <DialogFooter className="flex-row justify-between sm:justify-between">
          <span className="text-sm text-muted-foreground">
            {memories.length} memor{memories.length === 1 ? "y" : "ies"}
          </span>
          {memories.length > 0 && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setDeleteConfirm("all")}
            >
              Delete All
            </Button>
          )}
        </DialogFooter>

        {/* Edit dialog */}
        <Dialog open={!!editingMemory} onOpenChange={() => setEditingMemory(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Memory</DialogTitle>
            </DialogHeader>
            <textarea
              className="w-full min-h-[100px] rounded-md border bg-background p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring"
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
            />
            <DialogFooter>
              <Button variant="outline" onClick={() => setEditingMemory(null)}>
                Cancel
              </Button>
              <Button onClick={updateMemory}>Save</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete confirmation dialog */}
        <Dialog open={!!deleteConfirm} onOpenChange={() => setDeleteConfirm(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Confirm Delete</DialogTitle>
              <DialogDescription>
                {deleteConfirm === "all"
                  ? "Are you sure you want to delete ALL memories? This cannot be undone."
                  : "Are you sure you want to delete this memory? This cannot be undone."}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteConfirm(null)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={() =>
                  deleteConfirm === "all"
                    ? deleteAllMemories()
                    : deleteMemory(deleteConfirm!)
                }
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}
