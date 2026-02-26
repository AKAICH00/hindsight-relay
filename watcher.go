package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

// Watcher watches directories for .md file changes and calls callback.
type Watcher struct {
	mu       sync.Mutex
	fw       *fsnotify.Watcher
	debounce map[string]*time.Timer
	callback func(path string)
	dirs     []string
}

// NewWatcher creates a new Watcher with the given callback.
func NewWatcher(callback func(path string)) (*Watcher, error) {
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	return &Watcher{
		fw:       fw,
		debounce: make(map[string]*time.Timer),
		callback: callback,
	}, nil
}

// AddDir registers a directory for watching.
func (w *Watcher) AddDir(dir string) error {
	if err := w.fw.Add(dir); err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, d := range w.dirs {
		if d == dir {
			return nil
		}
	}
	w.dirs = append(w.dirs, dir)
	return nil
}

// Dirs returns currently watched directories.
func (w *Watcher) Dirs() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]string, len(w.dirs))
	copy(out, w.dirs)
	return out
}

// Start runs the event loop; blocks until ctx is cancelled.
func (w *Watcher) Start(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			_ = w.fw.Close()
			return
		case event, ok := <-w.fw.Events:
			if !ok {
				return
			}
			w.handleEvent(event)
		case err, ok := <-w.fw.Errors:
			if !ok {
				return
			}
			log.Printf("[watcher] error: %v", err)
		}
	}
}

func (w *Watcher) handleEvent(event fsnotify.Event) {
	if !strings.HasSuffix(event.Name, ".md") {
		return
	}

	switch {
	case event.Has(fsnotify.Create), event.Has(fsnotify.Write):
		path := event.Name
		w.mu.Lock()
		if t, ok := w.debounce[path]; ok {
			t.Reset(200 * time.Millisecond)
		} else {
			w.debounce[path] = time.AfterFunc(200*time.Millisecond, func() {
				w.mu.Lock()
				delete(w.debounce, path)
				w.mu.Unlock()
				w.callback(path)
			})
		}
		w.mu.Unlock()

	case event.Has(fsnotify.Rename), event.Has(fsnotify.Remove):
		log.Printf("[watcher] %s: %s (ignored)", event.Op, event.Name)
	}
}

// SaveState writes watched dirs to a JSON file.
func (w *Watcher) SaveState(path string) error {
	w.mu.Lock()
	dirs := make([]string, len(w.dirs))
	copy(dirs, w.dirs)
	w.mu.Unlock()

	data, err := json.MarshalIndent(dirs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadState reads watched dirs from a JSON file and re-watches them.
func (w *Watcher) LoadState(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // First run — no state file yet
		}
		return err
	}

	var dirs []string
	if err := json.Unmarshal(data, &dirs); err != nil {
		return err
	}

	for _, dir := range dirs {
		if _, err := os.Stat(dir); err != nil {
			log.Printf("[watcher] skipping missing dir %s: %v", dir, err)
			continue
		}
		if err := w.AddDir(dir); err != nil {
			log.Printf("[watcher] failed to re-watch %s: %v", dir, err)
		}
	}
	return nil
}

// watchedDirCount is a helper to get count without lock from outside.
func (w *Watcher) Count() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.dirs)
}

// isMarkdown checks extension.
func isMarkdown(path string) bool {
	return strings.EqualFold(filepath.Ext(path), ".md")
}
