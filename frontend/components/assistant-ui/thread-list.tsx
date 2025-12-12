"use client";

import { useState, type FC } from "react";
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAssistantState,
  useThreadListItemRuntime,
} from "@assistant-ui/react";
import { ArchiveIcon, PlusIcon, PencilIcon, Check, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Skeleton } from "@/components/ui/skeleton";

export const ThreadList: FC = () => {
  return (
    <ThreadListPrimitive.Root className="aui-root aui-thread-list-root flex flex-col items-stretch gap-1.5">
      <ThreadListNew />
      <ThreadListItems />
    </ThreadListPrimitive.Root>
  );
};

const ThreadListNew: FC = () => {
  return (
    <ThreadListPrimitive.New asChild>
      <Button
        className="aui-thread-list-new flex items-center justify-start gap-1 rounded-lg px-2.5 py-2 text-start hover:bg-muted data-active:bg-muted"
        variant="ghost"
      >
        <PlusIcon />
        New Thread
      </Button>
    </ThreadListPrimitive.New>
  );
};

const ThreadListItems: FC = () => {
  const isLoading = useAssistantState(({ threads }) => threads.isLoading);

  if (isLoading) {
    return <ThreadListSkeleton />;
  }

  return <ThreadListPrimitive.Items components={{ ThreadListItem }} />;
};

const ThreadListSkeleton: FC = () => {
  return (
    <>
      {Array.from({ length: 5 }, (_, i) => (
        <div
          key={i}
          role="status"
          aria-label="Loading threads"
          aria-live="polite"
          className="aui-thread-list-skeleton-wrapper flex items-center gap-2 rounded-md px-3 py-2"
        >
          <Skeleton className="aui-thread-list-skeleton h-[22px] flex-grow" />
        </div>
      ))}
    </>
  );
};

const ThreadListItem: FC = () => {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <ThreadListItemPrimitive.Root className="aui-thread-list-item group flex items-center gap-2 rounded-lg transition-all hover:bg-muted focus-visible:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none data-active:bg-muted">
      {isEditing ? (
        <ThreadListItemRename onClose={() => setIsEditing(false)} />
      ) : (
        <>
          <ThreadListItemPrimitive.Trigger className="aui-thread-list-item-trigger flex-grow px-3 py-2 text-start">
            <ThreadListItemTitle />
          </ThreadListItemPrimitive.Trigger>
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <ThreadListItemEdit onEdit={() => setIsEditing(true)} />
            <ThreadListItemArchive />
          </div>
        </>
      )}
    </ThreadListItemPrimitive.Root>
  );
};

const ThreadListItemTitle: FC = () => {
  return (
    <span className="aui-thread-list-item-title text-sm truncate">
      <ThreadListItemPrimitive.Title fallback="New Chat" />
    </span>
  );
};

const ThreadListItemEdit: FC<{ onEdit: () => void }> = ({ onEdit }) => {
  return (
    <TooltipIconButton
      className="aui-thread-list-item-edit size-6 p-0 text-foreground hover:text-primary"
      variant="ghost"
      tooltip="Rename thread"
      onClick={(e) => {
        e.stopPropagation();
        onEdit();
      }}
    >
      <PencilIcon className="size-3.5" />
    </TooltipIconButton>
  );
};

const ThreadListItemRename: FC<{ onClose: () => void }> = ({ onClose }) => {
  const runtime = useThreadListItemRuntime();
  const currentTitle = runtime.getState().title || "";
  const [title, setTitle] = useState(currentTitle);

  const handleSave = async () => {
    if (title.trim() && title !== currentTitle) {
      await runtime.rename(title.trim());
    }
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSave();
    } else if (e.key === "Escape") {
      onClose();
    }
  };

  return (
    <div className="flex items-center gap-1 flex-grow px-2 py-1">
      <Input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        onKeyDown={handleKeyDown}
        className="h-7 text-sm"
        autoFocus
        onClick={(e) => e.stopPropagation()}
      />
      <TooltipIconButton
        className="size-6 p-0 text-foreground hover:text-primary"
        variant="ghost"
        tooltip="Save"
        onClick={(e) => {
          e.stopPropagation();
          handleSave();
        }}
      >
        <Check className="size-3.5" />
      </TooltipIconButton>
      <TooltipIconButton
        className="size-6 p-0 text-foreground hover:text-muted-foreground"
        variant="ghost"
        tooltip="Cancel"
        onClick={(e) => {
          e.stopPropagation();
          onClose();
        }}
      >
        <X className="size-3.5" />
      </TooltipIconButton>
    </div>
  );
};

const ThreadListItemArchive: FC = () => {
  return (
    <ThreadListItemPrimitive.Archive asChild>
      <TooltipIconButton
        className="aui-thread-list-item-archive mr-2 size-6 p-0 text-foreground hover:text-primary"
        variant="ghost"
        tooltip="Archive thread"
      >
        <ArchiveIcon className="size-3.5" />
      </TooltipIconButton>
    </ThreadListItemPrimitive.Archive>
  );
};
