const DATABASE_NAME = "local-meetscribe-recordings";
const DATABASE_VERSION = 1;
const STORE_NAME = "pending-recordings";
const LATEST_RECORDING_KEY = "latest";
const OPERATION_TIMEOUT_MS = 2000;

interface StoredPendingRecording {
  version: 1;
  id: string;
  blob: Blob;
  name: string;
  type: string;
  size: number;
  lastModified: number;
  savedAt: number;
  retryAttempts: number;
}

export interface PendingRecording {
  id: string;
  file: File;
  savedAt: number;
  retryAttempts: number;
}

export async function savePendingRecording(
  file: File,
  id: string,
  retryAttempts = 0
): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    const completed = transactionComplete(transaction);
    transaction.objectStore(STORE_NAME).put(
      {
        version: 1,
        id,
        blob: file.slice(0, file.size, file.type),
        name: file.name,
        type: file.type,
        size: file.size,
        lastModified: file.lastModified,
        savedAt: Date.now(),
        retryAttempts
      } satisfies StoredPendingRecording,
      LATEST_RECORDING_KEY
    );
    await completed;
  } finally {
    database.close();
  }
}

export async function loadPendingRecording(): Promise<PendingRecording | null> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const completed = transactionComplete(transaction);
    const request = transaction.objectStore(STORE_NAME).get(LATEST_RECORDING_KEY);
    const [stored] = await Promise.all([
      requestResult<StoredPendingRecording | undefined>(request),
      completed
    ]);
    if (!isStoredPendingRecording(stored)) return null;

    return {
      id: stored.id,
      file: new File([stored.blob], stored.name, {
        type: stored.type || stored.blob.type,
        lastModified: stored.lastModified
      }),
      savedAt: stored.savedAt,
      retryAttempts: stored.retryAttempts
    };
  } finally {
    database.close();
  }
}

export async function updatePendingRecordingRetryAttempts(
  id: string,
  retryAttempts: number
): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    const completed = transactionComplete(transaction);
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(LATEST_RECORDING_KEY);
    request.onsuccess = () => {
      const current = request.result as StoredPendingRecording | undefined;
      if (current?.id === id) {
        store.put({ ...current, retryAttempts }, LATEST_RECORDING_KEY);
      }
    };
    await completed;
  } finally {
    database.close();
  }
}

export async function clearPendingRecording(id: string): Promise<void> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    const completed = transactionComplete(transaction);
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(LATEST_RECORDING_KEY);
    request.onsuccess = () => {
      const current = request.result as StoredPendingRecording | undefined;
      if (current?.id === id) store.delete(LATEST_RECORDING_KEY);
    };
    await completed;
  } finally {
    database.close();
  }
}

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is unavailable."));
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    let settled = false;
    const timeoutId = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error("IndexedDB open timed out."));
    }, OPERATION_TIMEOUT_MS);
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      reject(error);
    };
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => {
      if (settled) {
        request.result.close();
        return;
      }
      settled = true;
      window.clearTimeout(timeoutId);
      resolve(request.result);
    };
    request.onerror = () => fail(request.error || new Error("Could not open IndexedDB."));
    request.onblocked = () => fail(new Error("IndexedDB upgrade is blocked."));
  });
}

function transactionComplete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timeoutId = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      try {
        transaction.abort();
      } catch {
        // The transaction may already be finishing.
      }
      reject(new Error("IndexedDB transaction timed out."));
    }, OPERATION_TIMEOUT_MS);
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      reject(error);
    };
    transaction.oncomplete = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      resolve();
    };
    transaction.onerror = () =>
      fail(transaction.error || new Error("IndexedDB transaction failed."));
    transaction.onabort = () =>
      fail(transaction.error || new Error("IndexedDB transaction was aborted."));
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("IndexedDB request failed."));
  });
}

function isStoredPendingRecording(value: unknown): value is StoredPendingRecording {
  if (!value || typeof value !== "object") return false;
  const stored = value as Partial<StoredPendingRecording>;
  return (
    stored.version === 1 &&
    typeof stored.id === "string" &&
    stored.id.length > 0 &&
    stored.blob instanceof Blob &&
    typeof stored.name === "string" &&
    stored.name.length > 0 &&
    typeof stored.type === "string" &&
    typeof stored.size === "number" &&
    stored.size > 0 &&
    stored.blob.size === stored.size &&
    typeof stored.lastModified === "number" &&
    Number.isFinite(stored.lastModified) &&
    typeof stored.savedAt === "number" &&
    Number.isFinite(stored.savedAt) &&
    typeof stored.retryAttempts === "number" &&
    Number.isInteger(stored.retryAttempts) &&
    stored.retryAttempts >= 0
  );
}
